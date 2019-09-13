import logging
from loudml.api import (
    Service,
    Job,
    get_job_id,
)


class BucketService(Service):
    def __init__(
        self,
    ):
        super().__init__(prefix='/buckets')

    def read(
        self,
        bucket_name,
        from_date,
        to_date,
        bucket_interval,
        features,
    ):
        params = {
            'from': from_date,
            'to': to_date,
            'bucket_interval': bucket_interval,
            'features': features,
        }
        response = self._do_one(bucket_name, '_read', params)
        if response.ok:
            job_id = get_job_id(response)
            return Job(
                job_id,
                self._loud,
                name='read({})'.format(bucket_name),
                total=1,
            )
        else:
            logging.error(response.text)
            return None

    def write(
        self,
        bucket_name,
        points,
        batch_size,
        **kwargs
    ):
        params = kwargs
        for batch in [
            points[i:i+batch_size]
            for i in range(
                0, len(points), batch_size)
        ]:
            response = self._do_one(
                bucket_name, '_write', params, content=batch)
            if response.ok:
                job_id = get_job_id(response)
                yield Job(
                    job_id,
                    self._loud,
                    name='write({})'.format(bucket_name),
                    total=1,
                )
            else:
                logging.error(response.text)

    def clear(
        self,
        bucket_name,
    ):
        response = self._do_one(bucket_name, '_clear')
        return response
